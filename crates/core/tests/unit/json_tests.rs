// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for json in the NeMo Relay core crate.

use super::*;
use serde_json::json;

#[test]
fn test_merge_both_objects() {
    let a = Some(json!({"x": 1}));
    let b = Some(json!({"y": 2}));
    let merged = merge_json(a, b).unwrap();
    assert_eq!(merged, json!({"x": 1, "y": 2}));
}

#[test]
fn test_merge_one_none() {
    let a = Some(json!({"x": 1}));
    assert_eq!(merge_json(a.clone(), None), a);
    assert_eq!(merge_json(None, a.clone()), a);
}

#[test]
fn test_merge_both_none() {
    assert_eq!(merge_json(None, None), None);
}

#[test]
fn test_merge_overlapping_keys() {
    let a = Some(json!({"x": 1, "y": 2}));
    let b = Some(json!({"y": 99, "z": 3}));
    let merged = merge_json(a, b).unwrap();
    assert_eq!(merged, json!({"x": 1, "y": 99, "z": 3}));
}

#[test]
fn test_merge_non_object_a_object_b() {
    // When a is not an object but b is, b wins
    let a = Some(json!(42));
    let b = Some(json!({"key": "val"}));
    let merged = merge_json(a, b).unwrap();
    assert_eq!(merged, json!({"key": "val"}));
}

#[test]
fn test_merge_object_a_non_object_b() {
    // When a is object but b is not an object, b wins
    let a = Some(json!({"key": "val"}));
    let b = Some(json!("string_value"));
    let merged = merge_json(a, b).unwrap();
    assert_eq!(merged, json!("string_value"));
}

#[test]
fn test_merge_both_non_objects() {
    let a = Some(json!(42));
    let b = Some(json!("hello"));
    let merged = merge_json(a, b).unwrap();
    assert_eq!(merged, json!("hello"));
}

#[test]
fn test_merge_nested_objects() {
    // merge_json does shallow merge, not deep
    let a = Some(json!({"outer": {"inner_a": 1}}));
    let b = Some(json!({"outer": {"inner_b": 2}}));
    let merged = merge_json(a, b).unwrap();
    // b's "outer" replaces a's "outer" entirely (shallow merge)
    assert_eq!(merged, json!({"outer": {"inner_b": 2}}));
}

#[test]
fn test_merge_empty_objects() {
    let a = Some(json!({}));
    let b = Some(json!({}));
    let merged = merge_json(a, b).unwrap();
    assert_eq!(merged, json!({}));
}

#[test]
fn test_merge_a_none_b_non_object() {
    let merged = merge_json(None, Some(json!(42)));
    assert_eq!(merged, Some(json!(42)));
}

#[test]
fn test_merge_a_non_object_b_none() {
    let merged = merge_json(Some(json!([1, 2, 3])), None);
    assert_eq!(merged, Some(json!([1, 2, 3])));
}
